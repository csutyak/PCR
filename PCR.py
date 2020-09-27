import random
import pandas
import numpy as np
import matplotlib.pyplot as plt

INITIALRNA = "atgtttgt ttttcttgtt ttattgccac tagtctctag tcagtgtgtt aatcttacaa ccagaactca attaccccct gcatacacta attctttcac  acgtggtgtt tattaccctg acaaagtttt cagatcctca gttttacatt caactcaggacttgttctta cctttctttt ccaatgttac ttggttccat gctatacatg tctctgggaccaatggtact aagaggtttg ataaccctgt cctaccattt aatgatggtg tttattttgcttccactgag aagtctaaca taataagagg ctggattttt ggtactactt tagattcgaagacccagtcc ctacttattg ttaataacgc tactaatgtt gttattaaag tctgtgaatttcaattttgt aatgatccat ttttgggtgt ttattaccac aaaaacaaca aaagttggatggaaagtgag ttcagagttt attctagtgc gaataattgc acttttgaat atgtctctcagccttttctt atggaccttg aaggaaaaca gggtaatttc aaaaatctta gggaatttgtgtttaagaat attgatggtt attttaaaat atattctaag cacacgccta ttaatttagtgcgtgatctc cctcagggtt tttcggcttt agaaccattg gtagatttgc caataggtattaacatcact aggtttcaaa ctttacttgc tttacataga agttatttga ctcctggtgattcttcttca ggttggacag ctggtgctgc agcttattat gtgggttatc ttcaacctaggacttttcta ttaaaatata atgaaaatgg aaccattaca gatgctgtag actgtgcacttgaccctctc tcagaaacaa agtgtacgtt gaaatccttc actgtagaaa aaggaatctatcaaacttct aactttagag tccaaccaac agaatctatt gttagatttc ctaatattacaaacttgtgc ccttttggtg aagtttttaa cgccaccaga tttgcatctg tttatgcttggaacaggaag agaatcagca actgtgttgc tgattattct gtcctatata attccgcatcattttccact tttaagtgtt atggagtgtc tcctactaaa ttaaatgatc tctgctttactaatgtctat gcagattcat ttgtaattag aggtgatgaa gtcagacaaa tcgctccagggcaaactgga aagattgctg attataatta taaattacca gatgatttta caggctgcgttatagcttgg aattctaaca atcttgattc taaggttggt ggtaattata attacctgtatagattgttt aggaagtcta atctcaaacc ttttgagaga gatatttcaa ctgaaatctatcaggccggt agcacacctt gtaatggtgt tgaaggtttt aattgttact ttcctttacaatcatatggt ttccaaccca ctaatggtgt tggttaccaa ccatacagag tagtagtactttcttttgaa cttctacatg caccagcaac tgtttgtgga cctaaaaagt ctactaatttggttaaaaac aaatgtgtca atttcaactt caatggttta acaggcacag gtgttcttactgagtctaac aaaaagtttc tgcctttcca acaatttggc agagacattg ctgacactactgatgctgtc cgtgatccac agacacttga gattcttgac attacaccat gttcttttggtggtgtcagt gttataacac caggaacaaa tacttctaac caggttgctg ttctttatcaggatgttaac tgcacagaag tccctgttgc tattcatgca gatcaactta ctcctacttggcgtgtttat tctacaggtt ctaatgtttt tcaaacacgt gcaggctgtt taataggggctgaacatgtc aacaactcat atgagtgtga catacccatt ggtgcaggta tatgcgctagttatcagact cagactaatt ctcctcggcg ggcacgtagt gtagctagtc aatccatcattgcctacact atgtcacttg gtgcagaaaa ttcagttgct tactctaata actctattgccatacccaca aattttacta ttagtgttac cacagaaatt ctaccagtgt ctatgaccaagacatcagta gattgtacaa tgtacatttg tggtgattca actgaatgca gcaatcttttgttgcaatat ggcagttttt gtacacaatt aaaccgtgct ttaactggaa tagctgttgaacaagacaaa aacacccaag aagtttttgc acaagtcaaa caaatttaca aaacaccaccaattaaagat tttggtggtt ttaatttttc acaaatatta ccagatccat caaaaccaagcaagaggtca tttattgaag atctactttt caacaaagtg acacttgcag atgctggcttcatcaaacaa tatggtgatt gccttggtga tattgctgct agagacctca tttgtgcacaaaagtttaac ggccttactg ttttgccacc tttgctcaca gatgaaatga ttgctcaatacacttctgca ctgttagcgg gtacaatcac ttctggttgg acctttggtg caggtgctgcattacaaata ccatttgcta tgcaaatggc ttataggttt aatggtattg gagttacacagaatgttctc tatgagaacc aaaaattgat tgccaaccaa tttaatagtg ctattggcaaaattcaagac tcactttctt ccacagcaag tgcacttgga aaacttcaag atgtggtcaaccaaaatgca caagctttaa acacgcttgt taaacaactt agctccaatt ttggtgcaatttcaagtgtt ttaaatgata tcctttcacg tcttgacaaa gttgaggctg aagtgcaaattgataggttg atcacaggca gacttcaaag tttgcagaca tatgtgactc aacaattaattagagctgca gaaatcagag cttctgctaa tcttgctgct actaaaatgt cagagtgtgtacttggacaa tcaaaaagag ttgatttttg tggaaagggc tatcatctta tgtccttccctcagtcagca cctcatggtg tagtcttctt gcatgtgact tatgtccctg cacaagaaaagaacttcaca actgctcctg ccatttgtca tgatggaaaa gcacactttc ctcgtgaaggtgtctttgtt tcaaatggca cacactggtt tgtaacacaa aggaattttt atgaaccacaaatcattact acagacaaca catttgtgtc tggtaactgt gatgttgtaa taggaattgtcaacaacaca gtttatgatc ctttgcaacc tgaattagac tcattcaagg aggagttagataaatatttt aagaatcata catcaccaga tgttgattta ggtgacatct ctggcattaatgcttcagtt gtaaacattc aaaaagaaat tgaccgcctc aatgaggttg ccaagaatttaaatgaatct ctcatcgatc tccaagaact tggaaagtat gagcagtata taaaatggccatggtacatt tggctaggtt ttatagctgg cttgattgcc atagtaatgg tgacaattatgctttgctgt atgaccagtt gctgtagttg tctcaagggc tgttgttctt gtggatcctgctgcaaattt gatgaagacg actctgagcc agtgctcaaa ggagtcaaat tacattacacataa"
INITIALRNA = INITIALRNA.replace(" ", "")
lengthOfInitialRNA = len(INITIALRNA)
CYCLES = 20
RANDOM_E = 1000
RANDOM_E_NEGATIVE = -1000
RANDOM_BASE_FALLOFF = 2500

PRIMER = "gatgctgtccgtgatccaca"
REVERSEPRIMER = "cccgccgaggagaattagtc"

random.seed()

def compString(primerString):
	CompSequence = primerString
	CompSequence = CompSequence.replace("a", "T")
	CompSequence = CompSequence.replace("t", "a")
	CompSequence = CompSequence.replace("g", "C")
	CompSequence = CompSequence.replace("c", "g")
	CompSequence = CompSequence.replace("T", "t")
	CompSequence = CompSequence.replace("C", "c")
	return CompSequence


class DNAConvert:
	def __init__(self, DNAStrand, threeToFive, cutOff = False):
		self.DNAStrand = DNAStrand
		self.length = len(DNAStrand)
		self.threeToFive = threeToFive
		self.cutOff = cutOff

	def printStrand(self):
		print("perfect primer: ", self.primerPerfect())
		print("primer cutoff: ", self.primerCutOFF())
		print(self.DNAStrand)

	def findComp(self):
		CompSequence = compString(self.DNAStrand)

		if self.threeToFive:
			newStrandThreeToFive = False
		else:
			newStrandThreeToFive = True

		newStrand = DNAConvert(CompSequence, newStrandThreeToFive)
		return newStrand

	def greatPrimerConvert(self):
		self.primerCutOFF()
		if self.cutOff:
			return DNAConvert("", True, cutOff = True)
		if self.threeToFive:
			return self.primerConvert()
		else:
			return self.ReversePrimerConvert()

	def randomNumberGen(self):
		randomRange = random.randrange(RANDOM_E_NEGATIVE, RANDOM_E)
		randomNum = RANDOM_BASE_FALLOFF + randomRange
		return randomNum

	def primerConvert(self):
		#finds the index of the primer
		foundIndex = self.DNAStrand.find(PRIMER)
		#creates a substring from the index on
		newString = self.DNAStrand[foundIndex:self.randomNumberGen():]
		newStrand = DNAConvert(newString, self.threeToFive)
		return newStrand

	def ReversePrimerConvert(self):
		reverseString = self.DNAStrand[::-1]
		#finds the index of the primer
		foundIndex = reverseString.find(REVERSEPRIMER)
		#creates a substring from the index on
		newString = reverseString[foundIndex:self.randomNumberGen():]
		newString = newString[::-1]
		newStrand = DNAConvert(newString, self.threeToFive)
		return newStrand

	def primerPerfect(self):
		primerLength = len(PRIMER)
		compReversePrimer = compString(REVERSEPRIMER)
		compPrimer = compString(PRIMER)
		lengthOfString = len(self.DNAStrand)

		if(self.DNAStrand.find(PRIMER) == 0):
			if(self.DNAStrand.find(compReversePrimer[::-1]) == (lengthOfString - primerLength)):
				return True
			else:
				return False
		elif(self.DNAStrand.find(compPrimer) == 0):
			if(self.DNAStrand.find(REVERSEPRIMER[::-1]) == (lengthOfString - primerLength)):
				return True
			else:
				return False
		
		return False

	def primerCutOFF(self):
		primerLength = len(PRIMER)
		compReversePrimer = compString(REVERSEPRIMER)
		compPrimer = compString(PRIMER)
		lengthOfString = len(self.DNAStrand)

		if(self.DNAStrand.find(PRIMER) != -1):
			if(self.DNAStrand.find(compReversePrimer[::-1]) != -1):
				self.cutOff = False
				return False
			else:
				self.cutOff = True
				return True
		elif(self.DNAStrand.find(compPrimer) != -1):
			if(self.DNAStrand.find(REVERSEPRIMER[::-1]) != -1):
				self.cutOff = False
				return False
			else:
				self.cutOff = True
				return True

		self.cutOff = True
		return True

	def getLength(self):
		self.length = len(self.DNAStrand)
		return self.length

	def findGCCount(self):
		self.getLength()
		gcCtr = 0
		for x in range(self.length):
			if self.DNAStrand[x] == "g" or self.DNAStrand[x] == "c":
				gcCtr += 1

		return gcCtr

#loops

DNAArray = []
DNAStrandNum = 2
print("Cycle:  1")
initialDNA1 = DNAConvert(INITIALRNA, True)
initialDNA2 = initialDNA1.findComp()

DNAArray.append(initialDNA1)
DNAArray.append(initialDNA2)

i = 2
while i <= CYCLES:
	workingArray = []
	loopCtr = 0
	print("Cycle: " , i)
	ctr = 0
	while loopCtr < DNAStrandNum:
		workingArray.append(DNAArray[loopCtr].greatPrimerConvert())
		workingStrand = DNAArray[loopCtr].findComp()
		workingArray.append(workingStrand.greatPrimerConvert())
		loopCtr += 1


	#ending loop
	DNAArray = workingArray
	DNAStrandNum = DNAStrandNum * 2
	i += 1

print("Finished all cycles")

ctr = 0
emptyStandCtr = 0
minLength = 10000000
maxLength = 0
sumLengthOfStrings = 0
sumGC = 0
arrayOfLengths = []
perfectCopyCounter = 0
while ctr < DNAStrandNum:
	#test for min length
	if(DNAArray[ctr].primerPerfect()):
		perfectCopyCounter += 1

	#find min
	if(minLength > DNAArray[ctr].getLength()):
		minLength = DNAArray[ctr].getLength()

	#find Max
	if(maxLength < DNAArray[ctr].getLength()):
		maxLength = DNAArray[ctr].getLength()

	#find sum of all strings
	sumLengthOfStrings += DNAArray[ctr].getLength()

	#find sum of GC
	sumGC += DNAArray[ctr].findGCCount()
	
	arrayOfLengths.append(DNAArray[ctr].getLength())

	if(DNAArray[ctr].cutOff):
		emptyStandCtr += 1
	#DNAArray[ctr].printStrand()
	ctr += 1

#print number of fragments
NumberOfFragments = DNAStrandNum - emptyStandCtr
print("Number of Fragments: " , NumberOfFragments)
print("Min length: ", minLength)
print("Max length: ", maxLength)

average = round((sumLengthOfStrings / NumberOfFragments), 1)
print("Average length of DNA fragments: ", average)

GCContent = round(sumGC / sumLengthOfStrings, 3)
GCContent *= 100
print("GC content: ", GCContent, "%")

print("Perfect Copies: ", perfectCopyCounter)

n, bins, patches = plt.hist(arrayOfLengths, 20, density=0, facecolor='g', alpha=1)
plt.xlabel('Length of stand')
plt.ylabel('Frequency')
plt.title('Distribution of lengths')
plt.axis([0, maxLength, 0, NumberOfFragments])
plt.grid(True)
plt.show()

print("FINISHED")