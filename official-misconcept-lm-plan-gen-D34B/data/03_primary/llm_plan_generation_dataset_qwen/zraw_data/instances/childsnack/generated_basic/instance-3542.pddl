; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 942365

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 child30 - child
    bread465 bread470 - bread-portion
    content99 content274 - content-portion
    tray177 tray434 - tray
    table402 table424 table441 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray177 kitchen)
     (at tray434 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread470)
     (at_kitchen_content content99)
     (at_kitchen_content content274)
     (not_allergic_gluten child30)
     (not_allergic_gluten child313)
     (waiting child313 table402)
     (waiting child30 table441)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child313)
     (served child30)
    )
  )
)
