(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot152 shot200 shot19 - shot
      ingredient422 ingredient123 ingredient168 - ingredient
      cocktail316 - cocktail
      dispenser228 dispenser161 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot152)
  (ontable shot200)
  (ontable shot19)
  (dispenses dispenser228 ingredient422)
  (dispenses dispenser161 ingredient123)
  (dispenses dispenser365 ingredient168)
  (clean shaker307)
  (clean shot152)
  (clean shot200)
  (clean shot19)
  (empty shaker307)
  (empty shot152)
  (empty shot200)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail316 ingredient168)
  (cocktail-part2 cocktail316 ingredient123)
)
 (:goal
  (and
      (contains shot152 cocktail316)
      (contains shot200 ingredient123)
)))
