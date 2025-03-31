(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot480 shot31 - shot
      ingredient445 ingredient369 ingredient301 - ingredient
      cocktail356 - cocktail
      dispenser437 dispenser175 dispenser459 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot480)
  (ontable shot31)
  (dispenses dispenser437 ingredient445)
  (dispenses dispenser175 ingredient369)
  (dispenses dispenser459 ingredient301)
  (clean shaker129)
  (clean shot480)
  (clean shot31)
  (empty shaker129)
  (empty shot480)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient445)
  (cocktail-part2 cocktail356 ingredient369)
)
 (:goal
  (and
      (contains shot480 cocktail356)
)))
