; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 637291

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 child234 - child
    bread99 bread186 - bread-portion
    content496 content440 - content-portion
    tray484 tray359 tray140 tray180 - tray
    table399 table456 table359 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray359 kitchen)
     (at tray140 kitchen)
     (at tray180 kitchen)
     (at_kitchen_bread bread99)
     (at_kitchen_bread bread186)
     (at_kitchen_content content496)
     (at_kitchen_content content440)
     (no_gluten_bread bread99)
     (no_gluten_content content440)
     (allergic_gluten child234)
     (not_allergic_gluten child122)
     (waiting child122 table359)
     (waiting child234 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child122)
     (served child234)
    )
  )
)
