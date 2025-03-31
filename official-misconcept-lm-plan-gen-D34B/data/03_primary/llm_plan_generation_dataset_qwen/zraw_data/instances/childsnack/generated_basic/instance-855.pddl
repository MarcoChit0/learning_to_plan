; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 945745

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child361 child36 - child
    bread279 bread247 - bread-portion
    content113 content241 - content-portion
    tray307 tray110 - tray
    table62 table467 table157 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray110 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_bread bread247)
     (at_kitchen_content content113)
     (at_kitchen_content content241)
     (no_gluten_bread bread279)
     (no_gluten_content content241)
     (allergic_gluten child36)
     (not_allergic_gluten child361)
     (waiting child361 table62)
     (waiting child36 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child361)
     (served child36)
    )
  )
)
