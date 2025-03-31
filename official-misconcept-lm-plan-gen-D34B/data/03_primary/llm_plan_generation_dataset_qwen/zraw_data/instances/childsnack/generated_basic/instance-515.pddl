; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 361554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child330 child289 - child
    bread230 bread361 - bread-portion
    content115 content40 - content-portion
    tray18 tray384 - tray
    table244 table425 table224 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray384 kitchen)
     (at_kitchen_bread bread230)
     (at_kitchen_bread bread361)
     (at_kitchen_content content115)
     (at_kitchen_content content40)
     (not_allergic_gluten child330)
     (not_allergic_gluten child289)
     (waiting child330 table244)
     (waiting child289 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child330)
     (served child289)
    )
  )
)
