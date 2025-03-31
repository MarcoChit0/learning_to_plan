; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 955772

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child234 - child
    bread307 - bread-portion
    content317 - content-portion
    tray153 tray246 tray44 tray343 - tray
    table235 table91 table463 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray153 kitchen)
     (at tray246 kitchen)
     (at tray44 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread307)
     (at_kitchen_content content317)
     (not_allergic_gluten child234)
     (waiting child234 table463)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child234)
    )
  )
)
