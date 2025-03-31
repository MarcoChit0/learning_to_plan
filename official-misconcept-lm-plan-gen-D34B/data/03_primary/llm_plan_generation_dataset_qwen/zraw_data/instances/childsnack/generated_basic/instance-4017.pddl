; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 524159

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child86 child344 child253 - child
    bread40 bread150 bread118 - bread-portion
    content22 content197 content92 - content-portion
    tray58 - tray
    table258 table72 table274 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray58 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread118)
     (at_kitchen_content content22)
     (at_kitchen_content content197)
     (at_kitchen_content content92)
     (not_allergic_gluten child344)
     (not_allergic_gluten child253)
     (not_allergic_gluten child86)
     (waiting child86 table72)
     (waiting child344 table258)
     (waiting child253 table258)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child86)
     (served child344)
     (served child253)
    )
  )
)
