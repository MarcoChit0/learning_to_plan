(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot168 shot247 shot267 - shot
      ingredient77 ingredient313 ingredient399 ingredient429 - ingredient
      cocktail132 - cocktail
      dispenser92 dispenser59 dispenser331 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot168)
  (ontable shot247)
  (ontable shot267)
  (dispenses dispenser92 ingredient77)
  (dispenses dispenser59 ingredient313)
  (dispenses dispenser331 ingredient399)
  (dispenses dispenser131 ingredient429)
  (clean shaker260)
  (clean shot168)
  (clean shot247)
  (clean shot267)
  (empty shaker260)
  (empty shot168)
  (empty shot247)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient429)
  (cocktail-part2 cocktail132 ingredient77)
)
 (:goal
  (and
      (contains shot168 cocktail132)
      (contains shot247 cocktail132)
)))
