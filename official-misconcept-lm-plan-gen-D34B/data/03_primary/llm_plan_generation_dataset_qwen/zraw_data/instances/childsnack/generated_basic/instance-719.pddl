; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 684058

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child205 - child
    bread87 - bread-portion
    content251 - content-portion
    tray207 tray182 tray25 tray183 - tray
    table23 table145 table144 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray182 kitchen)
     (at tray25 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread87)
     (at_kitchen_content content251)
     (not_allergic_gluten child205)
     (waiting child205 table23)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child205)
    )
  )
)
