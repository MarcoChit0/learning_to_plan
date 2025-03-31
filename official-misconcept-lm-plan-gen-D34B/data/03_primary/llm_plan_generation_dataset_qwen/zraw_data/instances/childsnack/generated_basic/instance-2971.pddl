; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 321831

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 - child
    bread433 - bread-portion
    content166 - content-portion
    tray193 tray343 tray424 tray274 - tray
    table45 table250 table325 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at tray343 kitchen)
     (at tray424 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_content content166)
     (not_allergic_gluten child121)
     (waiting child121 table250)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child121)
    )
  )
)
