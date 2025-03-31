(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot28 shot186 shot75 - shot
      ingredient331 ingredient133 - ingredient
      cocktail448 - cocktail
      dispenser303 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot28)
  (ontable shot186)
  (ontable shot75)
  (dispenses dispenser303 ingredient331)
  (dispenses dispenser33 ingredient133)
  (clean shaker170)
  (clean shot28)
  (clean shot186)
  (clean shot75)
  (empty shaker170)
  (empty shot28)
  (empty shot186)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient133)
  (cocktail-part2 cocktail448 ingredient331)
)
 (:goal
  (and
      (contains shot28 cocktail448)
      (contains shot186 cocktail448)
)))
