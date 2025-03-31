(define (problem prob)
 (:domain barman)
 (:objects 
      shaker196 - shaker
      left right - hand
      shot116 shot435 - shot
      ingredient193 ingredient170 ingredient134 ingredient278 - ingredient
      cocktail1 - cocktail
      dispenser81 dispenser241 dispenser236 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker196)
  (ontable shot116)
  (ontable shot435)
  (dispenses dispenser81 ingredient193)
  (dispenses dispenser241 ingredient170)
  (dispenses dispenser236 ingredient134)
  (dispenses dispenser225 ingredient278)
  (clean shaker196)
  (clean shot116)
  (clean shot435)
  (empty shaker196)
  (empty shot116)
  (empty shot435)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker196 l0)
  (shaker-level shaker196 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient193)
  (cocktail-part2 cocktail1 ingredient278)
)
 (:goal
  (and
      (contains shot116 cocktail1)
)))
