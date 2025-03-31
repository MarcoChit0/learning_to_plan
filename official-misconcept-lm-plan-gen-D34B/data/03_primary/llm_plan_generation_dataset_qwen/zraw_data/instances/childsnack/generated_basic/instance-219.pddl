; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 384932

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 child486 - child
    bread294 bread418 - bread-portion
    content346 content436 - content-portion
    tray209 tray471 tray491 - tray
    table72 table437 table436 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray471 kitchen)
     (at tray491 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread418)
     (at_kitchen_content content346)
     (at_kitchen_content content436)
     (not_allergic_gluten child486)
     (not_allergic_gluten child382)
     (waiting child382 table436)
     (waiting child486 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child382)
     (served child486)
    )
  )
)
