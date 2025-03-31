(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot28 shot289 - shot
      ingredient242 ingredient305 ingredient350 ingredient171 - ingredient
      cocktail190 - cocktail
      dispenser8 dispenser417 dispenser158 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot28)
  (ontable shot289)
  (dispenses dispenser8 ingredient242)
  (dispenses dispenser417 ingredient305)
  (dispenses dispenser158 ingredient350)
  (dispenses dispenser28 ingredient171)
  (clean shaker497)
  (clean shot28)
  (clean shot289)
  (empty shaker497)
  (empty shot28)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient242)
  (cocktail-part2 cocktail190 ingredient305)
)
 (:goal
  (and
      (contains shot28 cocktail190)
)))
