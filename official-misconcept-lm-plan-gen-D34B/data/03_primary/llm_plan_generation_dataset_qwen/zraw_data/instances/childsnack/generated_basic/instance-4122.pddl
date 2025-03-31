; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 449300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 child46 - child
    bread370 bread92 - bread-portion
    content389 content56 - content-portion
    tray83 tray152 tray283 - tray
    table435 table457 table158 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray152 kitchen)
     (at tray283 kitchen)
     (at_kitchen_bread bread370)
     (at_kitchen_bread bread92)
     (at_kitchen_content content389)
     (at_kitchen_content content56)
     (not_allergic_gluten child46)
     (not_allergic_gluten child382)
     (waiting child382 table457)
     (waiting child46 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child382)
     (served child46)
    )
  )
)
