; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 223621

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child18 child370 - child
    bread425 bread271 - bread-portion
    content243 content387 - content-portion
    tray287 - tray
    table294 table457 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray287 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_bread bread271)
     (at_kitchen_content content243)
     (at_kitchen_content content387)
     (not_allergic_gluten child18)
     (not_allergic_gluten child370)
     (waiting child18 table457)
     (waiting child370 table294)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child18)
     (served child370)
    )
  )
)
