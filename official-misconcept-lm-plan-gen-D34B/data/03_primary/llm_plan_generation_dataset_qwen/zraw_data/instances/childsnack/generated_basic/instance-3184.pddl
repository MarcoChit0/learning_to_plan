; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 323055

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child239 child91 - child
    bread418 bread319 - bread-portion
    content307 content20 - content-portion
    tray239 tray472 - tray
    table374 table434 table109 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray239 kitchen)
     (at tray472 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_bread bread319)
     (at_kitchen_content content307)
     (at_kitchen_content content20)
     (not_allergic_gluten child239)
     (not_allergic_gluten child91)
     (waiting child239 table434)
     (waiting child91 table374)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child239)
     (served child91)
    )
  )
)
