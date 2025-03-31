; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 388920

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child320 child449 - child
    bread475 bread444 - bread-portion
    content51 content401 - content-portion
    tray381 tray485 tray427 - tray
    table67 table28 table394 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray485 kitchen)
     (at tray427 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread444)
     (at_kitchen_content content51)
     (at_kitchen_content content401)
     (not_allergic_gluten child449)
     (not_allergic_gluten child320)
     (waiting child320 table67)
     (waiting child449 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child320)
     (served child449)
    )
  )
)
