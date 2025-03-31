(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot222 shot65 shot52 - shot
      ingredient229 ingredient244 ingredient161 - ingredient
      cocktail48 - cocktail
      dispenser131 dispenser48 dispenser288 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot222)
  (ontable shot65)
  (ontable shot52)
  (dispenses dispenser131 ingredient229)
  (dispenses dispenser48 ingredient244)
  (dispenses dispenser288 ingredient161)
  (clean shaker129)
  (clean shot222)
  (clean shot65)
  (clean shot52)
  (empty shaker129)
  (empty shot222)
  (empty shot65)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient244)
  (cocktail-part2 cocktail48 ingredient161)
)
 (:goal
  (and
      (contains shot222 cocktail48)
      (contains shot65 cocktail48)
)))
