; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 994960

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 child38 - child
    bread178 bread243 - bread-portion
    content252 content436 - content-portion
    tray204 tray76 - tray
    table227 table95 table378 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray204 kitchen)
     (at tray76 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_bread bread243)
     (at_kitchen_content content252)
     (at_kitchen_content content436)
     (no_gluten_bread bread243)
     (no_gluten_content content252)
     (allergic_gluten child38)
     (not_allergic_gluten child54)
     (waiting child54 table95)
     (waiting child38 table227)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child54)
     (served child38)
    )
  )
)
