; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 170675

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child87 - child
    bread138 - bread-portion
    content397 - content-portion
    tray381 tray26 tray229 tray261 - tray
    table177 table410 table75 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray26 kitchen)
     (at tray229 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread138)
     (at_kitchen_content content397)
     (not_allergic_gluten child87)
     (waiting child87 table410)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child87)
    )
  )
)
