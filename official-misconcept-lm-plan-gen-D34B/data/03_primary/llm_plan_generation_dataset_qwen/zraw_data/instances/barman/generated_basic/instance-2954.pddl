(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot28 - shot
      ingredient265 ingredient15 ingredient241 ingredient109 - ingredient
      cocktail142 - cocktail
      dispenser200 dispenser77 dispenser287 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot28)
  (dispenses dispenser200 ingredient265)
  (dispenses dispenser77 ingredient15)
  (dispenses dispenser287 ingredient241)
  (dispenses dispenser152 ingredient109)
  (clean shaker206)
  (clean shot28)
  (empty shaker206)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient15)
  (cocktail-part2 cocktail142 ingredient109)
)
 (:goal
  (and
      (contains shot28 cocktail142)
)))
