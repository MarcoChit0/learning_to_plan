(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot60 - shot
      ingredient180 ingredient424 ingredient279 ingredient173 - ingredient
      cocktail30 - cocktail
      dispenser471 dispenser439 dispenser254 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot60)
  (dispenses dispenser471 ingredient180)
  (dispenses dispenser439 ingredient424)
  (dispenses dispenser254 ingredient279)
  (dispenses dispenser96 ingredient173)
  (clean shaker43)
  (clean shot60)
  (empty shaker43)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient173)
  (cocktail-part2 cocktail30 ingredient279)
)
 (:goal
  (and
      (contains shot60 cocktail30)
)))
