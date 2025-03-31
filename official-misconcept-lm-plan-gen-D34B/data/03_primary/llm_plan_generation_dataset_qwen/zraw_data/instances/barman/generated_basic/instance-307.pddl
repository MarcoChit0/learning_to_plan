(define (problem prob)
 (:domain barman)
 (:objects 
      shaker196 - shaker
      left right - hand
      shot289 shot76 - shot
      ingredient126 ingredient463 - ingredient
      cocktail1 - cocktail
      dispenser400 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker196)
  (ontable shot289)
  (ontable shot76)
  (dispenses dispenser400 ingredient126)
  (dispenses dispenser122 ingredient463)
  (clean shaker196)
  (clean shot289)
  (clean shot76)
  (empty shaker196)
  (empty shot289)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker196 l0)
  (shaker-level shaker196 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient126)
  (cocktail-part2 cocktail1 ingredient463)
)
 (:goal
  (and
      (contains shot289 cocktail1)
)))
