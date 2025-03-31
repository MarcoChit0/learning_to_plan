(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot445 shot476 shot67 - shot
      ingredient16 ingredient157 ingredient438 ingredient443 - ingredient
      cocktail30 - cocktail
      dispenser403 dispenser145 dispenser249 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot445)
  (ontable shot476)
  (ontable shot67)
  (dispenses dispenser403 ingredient16)
  (dispenses dispenser145 ingredient157)
  (dispenses dispenser249 ingredient438)
  (dispenses dispenser440 ingredient443)
  (clean shaker153)
  (clean shot445)
  (clean shot476)
  (clean shot67)
  (empty shaker153)
  (empty shot445)
  (empty shot476)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient16)
  (cocktail-part2 cocktail30 ingredient438)
)
 (:goal
  (and
      (contains shot445 cocktail30)
      (contains shot476 ingredient16)
)))
