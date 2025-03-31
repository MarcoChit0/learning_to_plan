(define (problem prob)
 (:domain barman)
 (:objects 
      shaker434 - shaker
      left right - hand
      shot104 shot23 shot34 - shot
      ingredient411 ingredient286 ingredient120 ingredient89 - ingredient
      cocktail141 - cocktail
      dispenser247 dispenser53 dispenser162 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker434)
  (ontable shot104)
  (ontable shot23)
  (ontable shot34)
  (dispenses dispenser247 ingredient411)
  (dispenses dispenser53 ingredient286)
  (dispenses dispenser162 ingredient120)
  (dispenses dispenser9 ingredient89)
  (clean shaker434)
  (clean shot104)
  (clean shot23)
  (clean shot34)
  (empty shaker434)
  (empty shot104)
  (empty shot23)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker434 l0)
  (shaker-level shaker434 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient286)
  (cocktail-part2 cocktail141 ingredient120)
)
 (:goal
  (and
      (contains shot104 cocktail141)
      (contains shot23 ingredient120)
)))
