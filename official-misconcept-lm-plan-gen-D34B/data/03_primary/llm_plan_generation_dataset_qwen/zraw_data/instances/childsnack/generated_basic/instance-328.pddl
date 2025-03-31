; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 209841

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child326 - child
    bread88 - bread-portion
    content30 - content-portion
    tray191 tray473 tray431 tray374 - tray
    table58 table244 table339 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray473 kitchen)
     (at tray431 kitchen)
     (at tray374 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_content content30)
     (not_allergic_gluten child326)
     (waiting child326 table244)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child326)
    )
  )
)
