; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 945885

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child286 child91 - child
    bread462 bread433 - bread-portion
    content20 content325 - content-portion
    tray353 tray324 tray323 tray258 - tray
    table460 table71 table163 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray353 kitchen)
     (at tray324 kitchen)
     (at tray323 kitchen)
     (at tray258 kitchen)
     (at_kitchen_bread bread462)
     (at_kitchen_bread bread433)
     (at_kitchen_content content20)
     (at_kitchen_content content325)
     (no_gluten_bread bread433)
     (no_gluten_content content325)
     (allergic_gluten child91)
     (not_allergic_gluten child286)
     (waiting child286 table460)
     (waiting child91 table460)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child286)
     (served child91)
    )
  )
)
