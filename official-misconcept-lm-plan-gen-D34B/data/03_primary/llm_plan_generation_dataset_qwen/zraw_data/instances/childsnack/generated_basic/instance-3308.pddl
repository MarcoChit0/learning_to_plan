; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 972896

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child172 - child
    bread259 bread207 - bread-portion
    content186 content278 - content-portion
    tray479 tray470 tray491 tray394 - tray
    table209 table455 table269 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at tray470 kitchen)
     (at tray491 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread207)
     (at_kitchen_content content186)
     (at_kitchen_content content278)
     (no_gluten_bread bread259)
     (no_gluten_content content278)
     (allergic_gluten child179)
     (not_allergic_gluten child172)
     (waiting child179 table269)
     (waiting child172 table209)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child179)
     (served child172)
    )
  )
)
