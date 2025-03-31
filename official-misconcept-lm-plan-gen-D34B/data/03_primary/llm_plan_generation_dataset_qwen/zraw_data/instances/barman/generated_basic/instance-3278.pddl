(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot79 shot25 - shot
      ingredient278 ingredient300 ingredient351 - ingredient
      cocktail120 - cocktail
      dispenser184 dispenser259 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot79)
  (ontable shot25)
  (dispenses dispenser184 ingredient278)
  (dispenses dispenser259 ingredient300)
  (dispenses dispenser323 ingredient351)
  (clean shaker375)
  (clean shot79)
  (clean shot25)
  (empty shaker375)
  (empty shot79)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient278)
  (cocktail-part2 cocktail120 ingredient351)
)
 (:goal
  (and
      (contains shot79 cocktail120)
)))
