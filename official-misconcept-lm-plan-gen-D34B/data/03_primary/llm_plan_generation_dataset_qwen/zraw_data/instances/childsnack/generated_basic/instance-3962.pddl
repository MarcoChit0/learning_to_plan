; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 867353

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 - child
    bread383 - bread-portion
    content415 - content-portion
    tray181 tray85 tray186 - tray
    table377 table150 table172 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at tray85 kitchen)
     (at tray186 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_content content415)
     (not_allergic_gluten child31)
     (waiting child31 table150)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child31)
    )
  )
)
