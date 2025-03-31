; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 816889

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 child390 - child
    bread195 bread323 - bread-portion
    content298 content58 - content-portion
    tray66 tray375 tray256 - tray
    table374 table418 table36 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at tray375 kitchen)
     (at tray256 kitchen)
     (at_kitchen_bread bread195)
     (at_kitchen_bread bread323)
     (at_kitchen_content content298)
     (at_kitchen_content content58)
     (not_allergic_gluten child390)
     (not_allergic_gluten child331)
     (waiting child331 table36)
     (waiting child390 table418)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child331)
     (served child390)
    )
  )
)
