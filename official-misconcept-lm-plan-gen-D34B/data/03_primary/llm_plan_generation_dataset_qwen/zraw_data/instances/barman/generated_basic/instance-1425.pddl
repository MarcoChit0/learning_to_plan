(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot152 shot485 - shot
      ingredient180 ingredient276 - ingredient
      cocktail97 - cocktail
      dispenser448 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot152)
  (ontable shot485)
  (dispenses dispenser448 ingredient180)
  (dispenses dispenser142 ingredient276)
  (clean shaker189)
  (clean shot152)
  (clean shot485)
  (empty shaker189)
  (empty shot152)
  (empty shot485)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient180)
  (cocktail-part2 cocktail97 ingredient276)
)
 (:goal
  (and
      (contains shot152 cocktail97)
)))
