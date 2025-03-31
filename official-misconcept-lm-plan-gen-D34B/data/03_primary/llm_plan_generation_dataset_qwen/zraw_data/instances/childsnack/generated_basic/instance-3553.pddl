; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 406109

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 child174 - child
    bread203 bread46 - bread-portion
    content15 content225 - content-portion
    tray143 tray367 - tray
    table223 table151 table207 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray143 kitchen)
     (at tray367 kitchen)
     (at_kitchen_bread bread203)
     (at_kitchen_bread bread46)
     (at_kitchen_content content15)
     (at_kitchen_content content225)
     (not_allergic_gluten child174)
     (not_allergic_gluten child163)
     (waiting child163 table151)
     (waiting child174 table207)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child163)
     (served child174)
    )
  )
)
