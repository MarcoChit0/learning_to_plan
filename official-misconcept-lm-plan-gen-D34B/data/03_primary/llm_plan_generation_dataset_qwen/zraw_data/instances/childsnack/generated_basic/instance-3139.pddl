; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 227094

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child319 child479 - child
    bread142 bread386 - bread-portion
    content241 content29 - content-portion
    tray84 tray467 - tray
    table353 table425 table494 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread142)
     (at_kitchen_bread bread386)
     (at_kitchen_content content241)
     (at_kitchen_content content29)
     (not_allergic_gluten child319)
     (not_allergic_gluten child479)
     (waiting child319 table353)
     (waiting child479 table494)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child319)
     (served child479)
    )
  )
)
