; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 91888

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 child377 - child
    bread475 bread300 - bread-portion
    content294 content240 - content-portion
    tray416 tray387 - tray
    table359 table332 table32 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray416 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread300)
     (at_kitchen_content content294)
     (at_kitchen_content content240)
     (no_gluten_bread bread475)
     (no_gluten_content content240)
     (allergic_gluten child204)
     (not_allergic_gluten child377)
     (waiting child204 table332)
     (waiting child377 table32)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child204)
     (served child377)
    )
  )
)
