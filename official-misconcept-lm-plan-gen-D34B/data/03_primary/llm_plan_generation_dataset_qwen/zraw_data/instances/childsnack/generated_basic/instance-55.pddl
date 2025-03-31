; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 983515

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 child449 - child
    bread305 bread457 - bread-portion
    content428 content274 - content-portion
    tray486 tray488 - tray
    table327 table103 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray488 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_bread bread457)
     (at_kitchen_content content428)
     (at_kitchen_content content274)
     (not_allergic_gluten child388)
     (not_allergic_gluten child449)
     (waiting child388 table103)
     (waiting child449 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child388)
     (served child449)
    )
  )
)
