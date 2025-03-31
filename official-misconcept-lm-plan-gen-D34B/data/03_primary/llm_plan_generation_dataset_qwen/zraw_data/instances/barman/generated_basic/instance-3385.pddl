(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot338 shot422 shot105 - shot
      ingredient293 ingredient394 ingredient246 - ingredient
      cocktail375 - cocktail
      dispenser344 dispenser427 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot338)
  (ontable shot422)
  (ontable shot105)
  (dispenses dispenser344 ingredient293)
  (dispenses dispenser427 ingredient394)
  (dispenses dispenser370 ingredient246)
  (clean shaker321)
  (clean shot338)
  (clean shot422)
  (clean shot105)
  (empty shaker321)
  (empty shot338)
  (empty shot422)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient293)
  (cocktail-part2 cocktail375 ingredient394)
)
 (:goal
  (and
      (contains shot338 cocktail375)
      (contains shot422 cocktail375)
)))
