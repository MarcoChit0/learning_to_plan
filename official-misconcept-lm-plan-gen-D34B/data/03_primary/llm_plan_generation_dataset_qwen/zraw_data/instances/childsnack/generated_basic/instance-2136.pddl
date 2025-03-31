; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 447742

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child35 - child
    bread446 bread320 - bread-portion
    content16 content364 - content-portion
    tray352 tray77 tray148 - tray
    table168 table491 table78 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray77 kitchen)
     (at tray148 kitchen)
     (at_kitchen_bread bread446)
     (at_kitchen_bread bread320)
     (at_kitchen_content content16)
     (at_kitchen_content content364)
     (no_gluten_bread bread446)
     (no_gluten_content content16)
     (allergic_gluten child194)
     (not_allergic_gluten child35)
     (waiting child194 table168)
     (waiting child35 table168)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child194)
     (served child35)
    )
  )
)
