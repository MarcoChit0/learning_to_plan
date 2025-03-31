; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 49043

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 child234 - child
    bread399 bread437 - bread-portion
    content464 content66 - content-portion
    tray429 tray189 - tray
    table186 table114 table371 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at tray189 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_bread bread437)
     (at_kitchen_content content464)
     (at_kitchen_content content66)
     (no_gluten_bread bread437)
     (no_gluten_content content464)
     (allergic_gluten child234)
     (not_allergic_gluten child148)
     (waiting child148 table114)
     (waiting child234 table186)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child148)
     (served child234)
    )
  )
)
