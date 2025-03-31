; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 518335

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child91 child374 - child
    bread25 bread321 - bread-portion
    content221 content284 - content-portion
    tray22 tray177 tray170 - tray
    table71 table373 table156 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray22 kitchen)
     (at tray177 kitchen)
     (at tray170 kitchen)
     (at_kitchen_bread bread25)
     (at_kitchen_bread bread321)
     (at_kitchen_content content221)
     (at_kitchen_content content284)
     (no_gluten_bread bread321)
     (no_gluten_content content284)
     (allergic_gluten child374)
     (not_allergic_gluten child91)
     (waiting child91 table156)
     (waiting child374 table156)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child91)
     (served child374)
    )
  )
)
