; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 594823

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child106 child128 - child
    bread330 bread219 bread28 - bread-portion
    content475 content85 content219 - content-portion
    tray68 - tray
    table142 table434 table104 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at_kitchen_bread bread330)
     (at_kitchen_bread bread219)
     (at_kitchen_bread bread28)
     (at_kitchen_content content475)
     (at_kitchen_content content85)
     (at_kitchen_content content219)
     (not_allergic_gluten child128)
     (not_allergic_gluten child106)
     (not_allergic_gluten child10)
     (waiting child10 table142)
     (waiting child106 table434)
     (waiting child128 table104)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child106)
     (served child128)
    )
  )
)
