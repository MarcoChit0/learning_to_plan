(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot207 shot281 shot121 - shot
      ingredient156 ingredient457 ingredient465 ingredient76 - ingredient
      cocktail229 - cocktail
      dispenser285 dispenser75 dispenser126 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot207)
  (ontable shot281)
  (ontable shot121)
  (dispenses dispenser285 ingredient156)
  (dispenses dispenser75 ingredient457)
  (dispenses dispenser126 ingredient465)
  (dispenses dispenser311 ingredient76)
  (clean shaker221)
  (clean shot207)
  (clean shot281)
  (clean shot121)
  (empty shaker221)
  (empty shot207)
  (empty shot281)
  (empty shot121)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient156)
  (cocktail-part2 cocktail229 ingredient457)
)
 (:goal
  (and
      (contains shot207 cocktail229)
      (contains shot281 cocktail229)
)))
