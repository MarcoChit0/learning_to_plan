; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 447952

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 child335 - child
    bread46 bread123 - bread-portion
    content473 content393 - content-portion
    tray142 tray244 tray162 - tray
    table85 table188 table322 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at tray244 kitchen)
     (at tray162 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_bread bread123)
     (at_kitchen_content content473)
     (at_kitchen_content content393)
     (not_allergic_gluten child335)
     (not_allergic_gluten child334)
     (waiting child334 table322)
     (waiting child335 table188)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child334)
     (served child335)
    )
  )
)
