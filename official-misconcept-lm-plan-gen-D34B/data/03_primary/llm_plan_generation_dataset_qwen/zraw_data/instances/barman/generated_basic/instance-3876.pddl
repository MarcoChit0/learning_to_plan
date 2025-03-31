(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot191 shot337 - shot
      ingredient237 ingredient219 ingredient164 - ingredient
      cocktail271 - cocktail
      dispenser242 dispenser418 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot191)
  (ontable shot337)
  (dispenses dispenser242 ingredient237)
  (dispenses dispenser418 ingredient219)
  (dispenses dispenser89 ingredient164)
  (clean shaker7)
  (clean shot191)
  (clean shot337)
  (empty shaker7)
  (empty shot191)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient237)
  (cocktail-part2 cocktail271 ingredient219)
)
 (:goal
  (and
      (contains shot191 cocktail271)
)))
