; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 878737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child228 child35 - child
    bread496 bread118 - bread-portion
    content129 content208 - content-portion
    tray114 tray99 tray316 - tray
    table276 table479 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at tray99 kitchen)
     (at tray316 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_bread bread118)
     (at_kitchen_content content129)
     (at_kitchen_content content208)
     (not_allergic_gluten child35)
     (not_allergic_gluten child228)
     (waiting child228 table196)
     (waiting child35 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child228)
     (served child35)
    )
  )
)
