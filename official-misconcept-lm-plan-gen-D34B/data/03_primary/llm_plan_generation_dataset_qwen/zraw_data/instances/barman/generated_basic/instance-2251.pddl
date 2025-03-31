(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot156 shot464 shot153 - shot
      ingredient301 ingredient174 ingredient201 ingredient304 - ingredient
      cocktail473 - cocktail
      dispenser158 dispenser101 dispenser192 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot156)
  (ontable shot464)
  (ontable shot153)
  (dispenses dispenser158 ingredient301)
  (dispenses dispenser101 ingredient174)
  (dispenses dispenser192 ingredient201)
  (dispenses dispenser225 ingredient304)
  (clean shaker153)
  (clean shot156)
  (clean shot464)
  (clean shot153)
  (empty shaker153)
  (empty shot156)
  (empty shot464)
  (empty shot153)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient201)
  (cocktail-part2 cocktail473 ingredient174)
)
 (:goal
  (and
      (contains shot156 cocktail473)
      (contains shot464 ingredient301)
)))
