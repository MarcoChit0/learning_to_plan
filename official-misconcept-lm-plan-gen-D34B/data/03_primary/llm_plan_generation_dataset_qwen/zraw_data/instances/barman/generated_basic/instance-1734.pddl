(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot353 shot440 - shot
      ingredient244 ingredient78 ingredient434 - ingredient
      cocktail103 - cocktail
      dispenser284 dispenser268 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot353)
  (ontable shot440)
  (dispenses dispenser284 ingredient244)
  (dispenses dispenser268 ingredient78)
  (dispenses dispenser21 ingredient434)
  (clean shaker278)
  (clean shot353)
  (clean shot440)
  (empty shaker278)
  (empty shot353)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail103 ingredient78)
  (cocktail-part2 cocktail103 ingredient434)
)
 (:goal
  (and
      (contains shot353 cocktail103)
)))
