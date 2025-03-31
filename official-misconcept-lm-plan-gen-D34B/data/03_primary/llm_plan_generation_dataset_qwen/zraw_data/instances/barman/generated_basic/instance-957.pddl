(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot143 - shot
      ingredient366 ingredient230 ingredient199 ingredient318 - ingredient
      cocktail1 - cocktail
      dispenser113 dispenser332 dispenser289 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot143)
  (dispenses dispenser113 ingredient366)
  (dispenses dispenser332 ingredient230)
  (dispenses dispenser289 ingredient199)
  (dispenses dispenser16 ingredient318)
  (clean shaker257)
  (clean shot143)
  (empty shaker257)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient199)
  (cocktail-part2 cocktail1 ingredient366)
)
 (:goal
  (and
      (contains shot143 cocktail1)
)))
