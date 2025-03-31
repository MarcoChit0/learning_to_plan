(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot227 shot363 shot149 - shot
      ingredient126 ingredient143 ingredient497 ingredient216 - ingredient
      cocktail1 - cocktail
      dispenser485 dispenser332 dispenser241 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot227)
  (ontable shot363)
  (ontable shot149)
  (dispenses dispenser485 ingredient126)
  (dispenses dispenser332 ingredient143)
  (dispenses dispenser241 ingredient497)
  (dispenses dispenser458 ingredient216)
  (clean shaker459)
  (clean shot227)
  (clean shot363)
  (clean shot149)
  (empty shaker459)
  (empty shot227)
  (empty shot363)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient143)
)
 (:goal
  (and
      (contains shot227 cocktail1)
      (contains shot363 cocktail1)
)))
