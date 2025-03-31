; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 602708

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child324 - child
    bread319 - bread-portion
    content394 - content-portion
    tray168 tray391 tray190 tray437 - tray
    table118 table393 table133 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray168 kitchen)
     (at tray391 kitchen)
     (at tray190 kitchen)
     (at tray437 kitchen)
     (at_kitchen_bread bread319)
     (at_kitchen_content content394)
     (not_allergic_gluten child324)
     (waiting child324 table393)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child324)
    )
  )
)
