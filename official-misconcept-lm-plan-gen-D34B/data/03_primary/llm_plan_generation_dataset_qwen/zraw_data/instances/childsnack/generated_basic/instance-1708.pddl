; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 274557

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 - child
    bread339 - bread-portion
    content393 - content-portion
    tray397 tray17 tray329 - tray
    table437 table281 table164 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at tray17 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread339)
     (at_kitchen_content content393)
     (not_allergic_gluten child168)
     (waiting child168 table281)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child168)
    )
  )
)
