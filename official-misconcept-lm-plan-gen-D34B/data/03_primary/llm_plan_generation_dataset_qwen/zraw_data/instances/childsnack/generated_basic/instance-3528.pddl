; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 67824

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child153 - child
    bread160 bread291 - bread-portion
    content312 content29 - content-portion
    tray175 tray465 - tray
    table177 table98 table215 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray465 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread291)
     (at_kitchen_content content312)
     (at_kitchen_content content29)
     (not_allergic_gluten child153)
     (not_allergic_gluten child38)
     (waiting child38 table215)
     (waiting child153 table177)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child38)
     (served child153)
    )
  )
)
